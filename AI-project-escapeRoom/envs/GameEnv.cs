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
    private int maxSteps = 2000;
    private int currentStep;
    public List<int> PlayerMove;

    public float place_the_box_good = 5;
    public float finish_reward = 10;

    public float droping_box_bad = -2;
    public float culide_with_wall = -1;
    public float repeating_actions = -1;
    public float time_panalty = -0.5f;
    public float max_steps_panalty = -5;

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
        game.player.heldBox != null ? 1.0f : 0.0f,
        game.IsPressed ? 1.0f : 0.0f,
        game.IsOpen ? 1.0f : 0.0f,
        (float)game.player.Velocity.Length()
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
            case 3: game.player.Grab(game.box); break; // Interact (e.g., pick up the box)
            case 4: game.player.DropHeldBox(); break; // Interact (e.g., drop the box)
        }

        PlayerMove.Add(action);
        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////


        //placing the box on the button
        if (game.box.Intersects(game.button) && game.player.heldBox == null)
        {
            reward += place_the_box_good; // Reward for placing the box on the button
        }

        //exiting the room finish goal
        if (game.IsPressed && IsOutOfBounds(game.player))
        {
            reward += finish_reward; // Reward for escaping the room
            IsDone = true;
        }

        //reward for geting closer to the goals
        if (game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null
        || game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null)
        {
            reward += 0.2f;
        }

        ///////////////////////////////
        // Penalties for incorrect behaviors//
        ///////////////////////////////
        //droping the box for no resone
        if (game.player.heldBox == null
        && !game.box.Intersects(game.button))
        {
            reward -= droping_box_bad; // Penalty for dropping the box for no reason
        }

        //culiding with the walls (not the ground)
        if (game.player.Intersects(game.walls[2])
        || game.player.Intersects(game.walls[3])
        || game.player.Intersects(game.walls[4]))
        {
            reward -= culide_with_wall; // Penalty for colliding with the walls
        }

        //repeating actions
        if (PlayerMove.Skip(PlayerMove.Count - 50).Distinct().Count() < 3)
        {
            reward -= repeating_actions;
        }

        //time penalty
        if (currentStep % 100 == 0)
        {
            reward -= time_panalty; // Penalty for taking too long
        }

        // Reset if out of bounds
        if ((IsOutOfBounds(game.player) || IsOutOfBounds(game.box)) && !game.IsPressed)
        {
            game.player.DropHeldBox();
            ResetPlayerAndBox();
        }
        //panelty for geting away from the goals
        if (!game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null
        || !game.IsMovingToward(game.button, game.lastPlayerPosition) && game.player.heldBox != null)
        {
            reward -= 0.1f;
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            if (!game.IsPressed && !IsOutOfBounds(game.player))
            {
                reward -= finish_reward; // Reward for escaping the room
            }
            reward -= max_steps_panalty; // Small penalty for exceeding maximum steps
            game.player.DropHeldBox();
            ResetPlayerAndBox();
            IsDone = true;
            currentStep = 0;
        }

        Thread.Sleep(1);
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