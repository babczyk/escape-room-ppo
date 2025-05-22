using System.Runtime.Serialization.Formatters;
using AI_project_escapeRoom; // Replace 'YourNamespace' with the actual namespace where Game1 is defined
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Linq;
using System.Diagnostics;
using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 3000;
    private int currentStep;
    public List<int> PlayerMove;

    public float place_the_box_good = 5;
    public float finish_reward = 10;

    public float droping_box_bad = -2;
    public float culide_with_wall = -2;
    public float repeating_actions = -2;
    public float time_panalty = -1f;
    public float max_steps_panalty = -7;

    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
        this.PlayerMove = new List<int>();
    }

    public Vector<float> GetState()
    {
        var stateValues = new List<float>
    {
        (float)(game.player.Position.X / game.widthLevel),
        (float)(game.player.Position.Y / game.groundLevel),
        (float)(game.box.Position.X / game.widthLevel),
        (float)(game.box.Position.Y / game.groundLevel),
        (float)(game.button.Position.X / game.widthLevel),
        (float)(game.button.Position.Y / game.groundLevel),
        (float)(game.door.Position.X / game.widthLevel),
        (float)(game.door.Position.Y / game.groundLevel),
        game.player.heldBox != null ? 1.0f : 0.0f,
        game.IsPressed ? 1.0f : 0.0f,
    };

        return Vector<float>.Build.DenseOfArray(stateValues.ToArray());
    }

    public (Vector<float>, float, bool) Step(int action)
    {
        currentStep++;
        float reward = 0;
        bool IsDone = false;

        // Apply action
        switch (action)
        {
            case 0: game.player.Move(new Vector2(-10, 0)); break; // Move left
            case 1: game.player.Move(new Vector2(10, 0)); break;  // Move right
            case 2: if (game.player.IsGrounded) game.player.ApplyForce(new Vector2(0, -250)); game.player.IsGrounded = false; break; // Jump
            case 3: game.player.Grab(game.box); break; // Grab box
            case 4: game.player.DropHeldBox(); break; // Drop box
        }

        PlayerMove.Add(action);

        ///////////////////////////////
        // Rewards
        ///////////////////////////////

        // for placing box correctly
        if (game.box.Intersects(game.button) && game.player.heldBox == null)
        {
            reward += 1f;
        }

        // for successfully exiting the room
        if (game.IsPressed && IsOutOfBounds(game.player))
        {
            reward += 2f;
            IsDone = true;
        }

        // for moving toward goal
        if ((game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null && (action == 0 || action == 1))
         || (game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null && (action == 0 || action == 1)))
        {
            reward += 0.15f;
        }

        // for pressing the button
        if (game.IsPressed)
        {
            reward += 0.3f;
        }

        // for picking up box
        if (game.player.heldBox != null)
        {
            reward += 0.1f;
        }

        // for droping the box correctly
        if (action == 4 && game.player.heldBox != null && game.box.Intersects(game.button))
        {
            reward += 2f;
        }

        ///////////////////////////////
        // Penalties
        ///////////////////////////////

        // for dropping box not on button
        if (game.player.heldBox == null && !game.box.Intersects(game.button) && action == 4)
        {
            reward -= 0.2f;
            Console.WriteLine("[PENALTY] Dropped box off button: -0.5");
        }

        // for colliding with walls
        if (game.player.Intersects(game.walls[2]) || game.player.Intersects(game.walls[3]) || game.player.Intersects(game.walls[4]))
        {
            reward -= 0.2f;
            Console.WriteLine("[PENALTY] Collided with wall: -0.1");
        }

        // time penalty every 100 steps
        if (currentStep % 100 == 0)
        {
            reward -= 0.1f;
        }

        // for moving away from goal
        if ((!game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null && (action == 0 || action == 1))
         || (!game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null && (action == 0 || action == 1)))
        {
            reward -= 0.1f;
            Console.WriteLine("[PENALTY] Moving away from goal: -0.1");
        }

        // for cheating
        if (IsOutOfBounds(game.player) && game.IsPressed == false
        || IsOutOfBounds(game.box) && game.IsPressed == false)
        {
            ResetPlayerAndBox();
            reward -= 0.1f;
        }


        // if max steps exceeded (failure)
        if (currentStep >= maxSteps)
        {
            reward -= 1f;
            game.player.DropHeldBox();
            ResetPlayerAndBox();
            IsDone = true;
            currentStep = 0;
        }

        Thread.Sleep(1);
        Console.WriteLine($"[TOTAL REWARD THIS STEP]: {reward}");

        return (GetState(), reward, IsDone);
    }

    // Helper methods
    private bool IsOutOfBounds(GameObject obj)
    {
        return obj.Position.X < game.cameraPosition.X || obj.Position.X > game.cameraPosition.X + game.ScreenWidth ||
               obj.Position.Y < game.cameraPosition.Y || obj.Position.Y > game.cameraPosition.Y + game.ScreenHeight;
    }

    public void ResetPlayerAndBox()
    {
        ResetPlayerPosition();
        ResetBoxPosition();
    }

    private void ResetPlayerPosition()
    {
        game.player.Position = new Vector2(100, game.groundLevel - game.player.Size.Y);
    }

    private void ResetBoxPosition()
    {
        game.box.Position = new Vector2(200, game.groundLevel - game.box.Size.Y);
    }

    public bool IsDone()
    {
        return game.IsOpen || currentStep >= maxSteps;
    }
}