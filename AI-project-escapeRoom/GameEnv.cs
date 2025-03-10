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

class GameEnvironment
{
    private Game1 game;
    private int maxSteps = 5000;
    private int currentStep;
    public GameEnvironment(Game1 game)
    {
        this.game = game;
        this.currentStep = 0;
    }

    public double[] GetState()
    {
        // Example state representation
        return new double[]
        {
            game.player.Position.X / game.widthLevel,
            game.player.Position.Y / game.groundLevel,
            game.box.Position.X / game.widthLevel,
            game.box.Position.Y / game.groundLevel,
            game.IsPressed ? 1.0 : 0.0,
            game.IsOpen ? 1.0 : 0.0
        };
    }

    public (double[], int, bool) Step(int action)
    {
        currentStep++;
        int reward = 0;
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

        ///////////////////////////////
        // Rewards for key objectives//
        ///////////////////////////////

        // Reward for picking up the box
        if (game.player.heldBox != null && game.player.heldBox == game.box)
        {
            reward += 20; // Reward for picking up the box
        }

        // Reward for moving toward the box
        if (game.IsMovingToward(game.box, game.lastPlayerPosition) && !game.IsPressed
        && game.player.heldBox == null) // Only reward if the box is not held
        {
            reward += 7; // Encourage moving toward the box
        }
        else if (!game.IsMovingToward(game.box, game.lastPlayerPosition) && game.player.heldBox == null)
        {
            reward -= 5; // Small penalty for moving away from the box
        }

        // Reward for moving toward the button while holding the box
        if (game.player.heldBox != null && game.IsMovingToward(game.button, game.lastPlayerPosition))
        {
            reward += 20; // Encourage moving toward button while holding the box
        }
        else if (game.player.heldBox != null && !game.IsMovingToward(game.button, game.lastPlayerPosition))
        {
            reward -= 5; // Small penalty for moving away from the button
        }

        // Reward for placing the box on the button
        if (game.box.Intersects(game.button) && !game.previousBoxState)
        {
            reward += 100; // Large reward for placing the box on the button
            game.IsPressed = true;
            game.previousBoxState = true;
        }

        // Reward for moving toward the door after pressing the button
        if (game.IsPressed && game.IsMovingToward(game.door, game.lastPlayerPosition))
        {
            reward += 10;
        }
        else if (game.IsPressed && !game.IsMovingToward(game.door, game.lastPlayerPosition))
        {
            reward -= 5;
        }

        // Reward for reaching the door after pressing the button
        if (game.IsPressed && game.player.Intersects(game.door))
        {
            reward += 200; // Large reward for reaching the door after pressing the button
            IsDone = true;
        }

        ///////////////////////////////
        // Penalties for incorrect behaviors//
        ///////////////////////////////

        // Penalty for reaching the door without pressing the button
        if (!game.box.Intersects(game.button) && game.player.Intersects(game.door))
        {
            reward -= 20; // Penalty for reaching the door without pressing the button
        }

        // Reset if out of bounds
        if (IsOutOfBounds(game.player) || IsOutOfBounds(game.box))
        {
            if (game.box.Intersects(game.button) && IsOutOfBounds(game.player))
            {
                reward += 100; // Reward for escaping the room
                IsDone = true;
            }
            else
            {
                ResetPlayerAndBox();
            }
        }

        // Maximum steps penalty
        if (currentStep >= maxSteps)
        {
            reward -= 10; // Small penalty for exceeding maximum steps
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

    private void ResetPlayerAndBox()
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